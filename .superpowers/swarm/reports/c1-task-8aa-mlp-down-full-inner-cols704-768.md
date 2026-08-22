# C1 task 8aa — MLP down full-inner cols704:768

## Scope
Implemented bounded Llama layer0 MLP down projection full-inner primitive-chain block for output cols704:768 only.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full hidden-width/full layer0 acceptance, and makes no Qwen claim. Runtime/fake-bridge drift markers for cols704:768 were initially seeded from the nearest prior chain placeholders and repaired after supervisor hardware captured observed values.

## RED evidence
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle -q`
  - Result: exit 1; `1 failed, 11 passed`; failure was `FileNotFoundError` for `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_fixtures.npz`.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols704_768_chain -q`
  - Result: exit 1; runner rejected `layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain` as unsupported with `failure_stage: primitive_chain_request`, `wrapper_exit_status: 2`.

## Changed files
- `native_r9700/ref_fixtures.py`
  - Added cols704:768 to layer0 MLP down full-inner fixture generation and split final/chunk fixture emission.
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Added deterministic schema/digest entries for cols704:768 final and chunk NPZs.
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_chunk3_fixtures.npz`
- `native_r9700/c1_primitive_bridge.cpp`
  - Added cols704:768 metadata, dot2 row-pair/column-packed model weight bytes, expected fp32 bytes, chain spec, run function, and bridge dispatch.
- `native_r9700/runtime.h`
  - Added cols704:768 runtime constants/digests.
- `native_r9700/runtime.cpp`
  - Added wrapper support/marker validation for cols704:768; supervisor hardware later repaired cols704:768 drift markers to observed chain-specific values.
- `native_r9700/runner.cpp`
  - Added help exposure for `layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain`.
- `tests/native_r9700/test_ref_fixtures.py`
  - Added cols704:768 fixture oracle case and schema file expectations.
- `tests/native_r9700/test_runtime_contract.py`
  - Added cols704:768 constants, marker derivation, runner help coverage, embedded operand layout coverage, and primitive-chain wrapper test.
- `.superpowers/swarm/reports/c1-task-8aa-mlp-down-full-inner-cols704-768.md`

## Deterministic digests
- Final fixture: `bd14dc7b032e7c2fe2340743cbf360339f514c7ca0f938ea89665c473098f22e`.
- Expected fp32 array: `db8c4b0630d95561e8cc3dfebfbe2a3d0aa13b14116dbaaeb4fcf24efc45de0a`.
- Chunk0 fixture: `ca5e6fd96d0a1e765db4ed521db6a76554f5f3859b2d1b3e17e9550db211788b`.
- Chunk1 fixture: `4b4d5b99c2b2eac2dac098749887fbe928216779bf389b9e66f32809974f3643`.
- Chunk2 fixture: `e47b23b42c582e036401e224d7bd4287a174b76069181e626695f1075688115f`.
- Chunk3 fixture: `8a57c1af05212d59cb0c9ec0bb7ad63df89e670ccd2ecb7cdaed4c164a14fc20`.

## GREEN verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: exit 0; wrote 76 fixture files including cols704:768 final/chunk NPZs and `fixtures_schema.json`.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_layer0_mlp_down_cols704_768_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols704_768_chain -q`
  - Result: exit 0; `16 passed in 20.85s`.
- `mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge`
  - Result: exit 0; no compiler output.
- `mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exit 0; no compiler output.

## Suggested supervisor hardware command
```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain
```

Or from a clean focused build:
```sh
mkdir -p build/native-r9700-runtime && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && \
  build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain
```

## Supervisor verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols704_768_chain or layer0_mlp_down_cols704_768_embedded_operands_use_kernel_layouts' -q`
  - Result: exit 0; `2 passed, 126 deselected in 14.76s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `tolerance: fp32_abs<=2e-4_or_ulp<=64`, `max_abs_diff=6.6280364990234375e-05`, `max_ulp_diff=6092`, `mismatch_count=0`, `byte_mismatch_count=458`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain-2026-08-20T20:44:34Z.log`.

## Remaining blockers
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains open.
- No full hidden-width layer0/native prefill/full attention width/Qwen acceptance is claimed.
