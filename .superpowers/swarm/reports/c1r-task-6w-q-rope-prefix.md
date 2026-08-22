# C1R-6w Q RoPE prefix-token full-head hardware proof

Status: proof-only hardware primitive chain complete.

## Decision

Add `layer0_q_rope_tokens0_5_head0_full_head_chain` as the Q prompt-prefix analogue of the existing K prefix full-head RoPE chain.

Reuse the proven `fp16_rope_split_half_layer0_q_pairs8` primitive for 20 resident dispatches: prompt tokens `0:5` × pair chunks `0:8`, `8:16`, `16:24`, `24:32`.

`native_prefill_acceptance` remains `open`; this is a proof-only slice for layer0 head0 Q RoPE over all prompt-0 prefix tokens, not full prefill acceptance.

## Files changed

- `native_r9700/ref_fixtures.py`
  - Emits `layer0_q_rope_tokens0_5_head0_full_head_{input_fp16,cos_fp32,sin_fp32,expected_fp16}`.
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
  - Regenerated compact layer trace fixture with Q prefix RoPE arrays.
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Records Q prefix RoPE array schema and digest metadata.
- `tests/native_r9700/test_ref_fixtures.py`
  - Adds Q prefix expected specs to the global key set.
  - Verifies split-half oracle equivalence, token0 zero-sin behavior, later-token nonzero sin, and token1 equality with the existing Q token1 full-head fixture.
- `native_r9700/runtime.h`
  - Adds public marker constants for `kLayer0QRopeTokens05Head0FullHeadChain*`.
- `native_r9700/runtime.cpp`
  - Recognizes Q prefix RoPE, includes it in supported chains, and validates generic plus 20 per-stage bridge markers.
- `native_r9700/c1_primitive_bridge.cpp`
  - Adds embedded Q prefix input/expected bytes, log/finish/run path, and bridge dispatch route.
- `native_r9700/runner.cpp`
  - Lists `layer0_q_rope_tokens0_5_head0_full_head_chain` in `--help`.
- `tests/native_r9700/test_runtime_contract.py`
  - Adds wrapper success and missing-`source_arrays` rejection coverage for the Q prefix chain.
- `docs/tasks/native-r9700-producer/validation-commands.md`
  - Records focused tests, compile, hardware proof, markers, and acceptance decision.

## Fixture evidence

Generated with:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
```

Measured fixture sizes/digests after Q prefix regeneration:

- `layer_trace_fixtures.npz`: 236,770 bytes; SHA256 `84b74c99cec260fd4196c2ff496a35566fb14413743f6e5b1f4b0ad5a1fdad45`
- `layer_trace_full_inner_projection_fixtures.npz`: 453,508 bytes; SHA256 `b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c`
- `layer_trace_q_full_inner_projection_fixtures.npz`: 227,045 bytes; SHA256 `63fee1efc814355717f47278d0cf2b2f617b66d948e1621ea6b8e46057ad1ce8`
- `fixtures_schema.json`: 20,706 bytes; SHA256 `55e53b5589909c4f848199da83a6f56d9f931b030b2841c178fe0c3dd4f91d3b`

Q prefix expected chunked fp16 stream:

- source array shape: `(5, 2, 32)` fp16
- chunked output bytes: 640
- SHA256: `b8d6efd4a75a399602ab4ca6e69d3192c066e24fceb153852699064a513436d8`

## Verification

Focused fixture/runtime tests:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_q_rope_tokens0_5_head0_full_head_fixture_matches_split_half_oracle \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_rope_tokens0_5_head0_full_head_chain \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_q_rope_prefix_source_arrays_marker \
  -q
```

Result: initial `4 passed in 3.68s`; post-review focused set `5 passed in 5.62s`.

Bridge/runner compile:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_c1_primitive_bridge && \
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Result: exit 0, no compiler output.

Direct bridge proof:

```sh
build/native-r9700-runtime/native_r9700_c1_primitive_bridge --primitive-chain layer0_q_rope_tokens0_5_head0_full_head_chain
```

Result: exit 0; important markers: `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `max_abs_diff: 0`, `max_ulp_diff: 0`, `mismatch_count: 0`, `byte_mismatch_count: 0`.

Hardware wrapper proof:

```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_q_rope_tokens0_5_head0_full_head_chain
```

Result: initial exit 0; post-review exit 0; latest log `logs/c1-runner-primitive-chain-proof-layer0_q_rope_tokens0_5_head0_full_head_chain-2026-08-19T21:49:36Z.log`.

Important markers:

- `primitive_chain_proof_wrapper_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `mismatch_count: 0`
- `max_abs_diff: 0`
- `max_ulp_diff: 0`
- `byte_mismatch_count: 0`
- `chain_stage_count: 20`
- `rope_token_count: 5`
- `rope_pair_chunks_per_token: 4`
- `expected_chunked_fp16_sha256: b8d6efd4a75a399602ab4ca6e69d3192c066e24fceb153852699064a513436d8`

## Review notes

Targeted reviewer `C1R6wQRopeReview` found two Minor issues: missing runner help discoverability and a stale C1R-6v validation heading. Supervisor fixed both, re-ran focused tests, recompiled, re-ran Q prefix hardware proof, full native regression, and whitespace gate.

Full native regression after review fixes: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `192 passed, 2 warnings in 98.99s`.

Whitespace gate after review fixes: `git diff --check` -> exit 0, no output.
